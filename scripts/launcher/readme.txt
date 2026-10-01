scripts/launcher/ —— 启动/停止脚本
================================================

作用：Windows/Linux 快速启动、停止与重建命令。

  run.bat / run.sh             启动完整系统（镜像不存在时自动构建）
  rebuild-run.bat              依赖或构建产物变化后重建再启动
  docker-run.bat / docker-run.sh  Docker 方式启动
  stop.bat / stop.sh           停止服务（不删除数据）
