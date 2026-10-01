frontend/ —— React + TypeScript + Vite 前端
================================================

作用：浏览器端知识库管理与可信问答界面。

  index.html / vite.config.ts / tsconfig*.json   构建配置
  package.json                依赖与脚本（npm ci && npm run build）
  src/                        源码
    App.tsx / main.tsx         应用入口
    pages/                    页面（文档库、可信问答、系统状态等）
    components/               组件
    api/                     后端 API 调用封装
    state/                   状态管理
    utils/ types/ assets/     工具、类型、静态资源
    styles.css                全局样式
  public/brand/               品牌图标资源
