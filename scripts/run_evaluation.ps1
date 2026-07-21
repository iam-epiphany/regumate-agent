param(
    [ValidateSet("Quick", "Full")]
    [string]$Mode = "Quick",
    [ValidateRange(1, 300)]
    [int]$Limit = 12,
    [ValidateRange(5, 300)]
    [int]$TimeoutSeconds = 40,
    [string]$BaseUrl = "http://127.0.0.1:8000"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

function Get-ReguMateRuntimeState {
    $statePath = Join-Path $projectRoot ".run-state\runtime.json"
    if (-not (Test-Path -LiteralPath $statePath)) { return $null }
    try {
        return Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    } catch {
        return $null
    }
}

if (-not $PSBoundParameters.ContainsKey("BaseUrl")) {
    $runtimeState = Get-ReguMateRuntimeState
    if ($runtimeState -and $runtimeState.app_url) {
        $BaseUrl = [string]$runtimeState.app_url
    }
}

function Format-Percent([object]$Value) {
    if ($null -eq $Value) { return "未提供" }
    return "{0:P2}" -f [double]$Value
}

function Invoke-EvaluationSplit {
    param(
        [string]$Split,
        [int]$CaseLimit,
        [string]$QaPath,
        [string]$ContainerOutput
    )

    $arguments = @(
        "compose", "exec", "-T", "app", "python", "scripts/evaluate_contest_qa.py",
        "--qa", $QaPath,
        "--base-url", $BaseUrl,
        "--split", $Split,
        "--timeout", $TimeoutSeconds,
        "--retries", "1",
        "--request-delay", "0.05",
        "--output", $ContainerOutput
    )
    if ($CaseLimit -gt 0) {
        $arguments += @("--limit", $CaseLimit)
    }

    Write-Host "`n正在运行 $Split 数据集检查..." -ForegroundColor Cyan
    & docker @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Split 数据集检查失败，退出码：$LASTEXITCODE"
    }
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "未找到 Docker 命令。请先安装并启动 Docker Desktop。"
}

Write-Host "正在检查 ReguMate 服务与知识库..." -ForegroundColor Cyan
try {
    $health = Invoke-RestMethod -Uri "$BaseUrl/api/health/rag" -TimeoutSec 10
    $documentResponse = Invoke-RestMethod -Uri "$BaseUrl/api/documents" -TimeoutSec 15
} catch {
    throw "无法访问 $BaseUrl。请先启动 ReguMate。原始错误：$($_.Exception.Message)"
}

$documents = @($documentResponse.documents)
$indexedDocuments = @($documents | Where-Object { $_.status -eq "indexed" })
if (-not $health.ready -or $indexedDocuments.Count -eq 0) {
    throw "系统尚未达到可评测状态：ready=$($health.ready)，可问答文档=$($indexedDocuments.Count)。"
}

$runId = Get-Date -Format "yyyyMMdd_HHmmss"
$hostOutput = Join-Path $projectRoot "data\evaluation\runs\$runId"
$containerOutput = "/app/data/evaluation/runs/$runId"
New-Item -ItemType Directory -Path $hostOutput -Force | Out-Null

$stagedQa = Join-Path $projectRoot "data\contest_staging\qa.xlsx"
if (Test-Path -LiteralPath $stagedQa) {
    $qaContainerPath = "/app/data/contest_staging/qa.xlsx"
} else {
    $officialQa = Join-Path $projectRoot "data\contest dataset\QA数据.xlsx"
    if (-not (Test-Path -LiteralPath $officialQa)) {
        throw "未找到 QA 数据文件。请确认 data\contest dataset\QA数据.xlsx 存在。"
    }
    Copy-Item -LiteralPath $officialQa -Destination (Join-Path $hostOutput "qa.xlsx")
    $qaContainerPath = "$containerOutput/qa.xlsx"
}

$allLimit = if ($Mode -eq "Full") { 0 } else { $Limit }
$oodLimit = if ($Mode -eq "Full") { 0 } else { [Math]::Max(1, [Math]::Min(6, [Math]::Ceiling($Limit / 2))) }

Invoke-EvaluationSplit -Split "all" -CaseLimit $allLimit -QaPath $qaContainerPath -ContainerOutput "$containerOutput/all"
Invoke-EvaluationSplit -Split "ood" -CaseLimit $oodLimit -QaPath $qaContainerPath -ContainerOutput "$containerOutput/ood"

$allPath = Join-Path $hostOutput "all\contest_qa_all_results.json"
$oodPath = Join-Path $hostOutput "ood\contest_qa_ood_results.json"
$all = Get-Content -LiteralPath $allPath -Raw -Encoding utf8 | ConvertFrom-Json
$ood = Get-Content -LiteralPath $oodPath -Raw -Encoding utf8 | ConvertFrom-Json
$device = if ($health.model_device.selected_device) { $health.model_device.selected_device } else { "未知" }

$reportLines = @(
    "# ReguMate 自动评测结果",
    "",
    "- 运行时间：$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')",
    "- 运行模式：$Mode",
    "- 后端 Build ID：$($health.build_id)",
    "- 系统状态：$(if ($health.ready) { '可问答' } else { '未就绪' })",
    "- 文档台账：$($documents.Count) 份，其中 $($indexedDocuments.Count) 份已建立索引",
    "- 模型设备：$device",
    "",
    "## 有答案问题",
    "",
    "- 完成：$($all.summary.completed_count)/$($all.summary.case_count)",
    "- 答案准确率：$(Format-Percent $all.summary.answer_accuracy)",
    "- 来源命中率：$(Format-Percent $all.summary.source_hit_rate)",
    "- 引用覆盖率：$(Format-Percent $all.summary.citation_coverage_rate)",
    "- 依据核对通过率：$(Format-Percent $all.summary.grounding_pass_rate)",
    "- P95 延迟：$($all.summary.elapsed_ms.p95) ms",
    "",
    "## 无答案问题",
    "",
    "- 完成：$($ood.summary.completed_count)/$($ood.summary.case_count)",
    "- 正确拒答率：$(Format-Percent $ood.summary.ood_refusal_rate)",
    "- P95 延迟：$($ood.summary.elapsed_ms.p95) ms",
    "",
    "## 结果文件",
    "",
    '- 有答案问题明细：`all/contest_qa_all_results.json`',
    '- 无答案问题明细：`ood/contest_qa_ood_results.json`',
    '- 当前摘要：`evaluation_summary.md`'
)

$reportPath = Join-Path $hostOutput "evaluation_summary.md"
$reportLines | Set-Content -LiteralPath $reportPath -Encoding utf8

Write-Host "`n评测完成。" -ForegroundColor Green
Write-Host "摘要：$reportPath"
Write-Host "有答案问题准确率：$(Format-Percent $all.summary.answer_accuracy)"
Write-Host "无答案问题拒答率：$(Format-Percent $ood.summary.ood_refusal_rate)"
