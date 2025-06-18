module.exports = {
  apps: [{
    name: "fastapi-chatbot",
    script: "/app/chatbot/run.sh",
    interpreter: "bash",
    cwd: "/app/chatbot",
    watch: false,
    env: {
      NODE_ENV: "production",
      PYTHONUNBUFFERED: "1"
    },
    error_file: "/app/logs/error.log",
    out_file: "/app/logs/out.log",
    log_date_format: "YYYY-MM-DD HH:mm:ss",
    merge_logs: true,
    time: true,
    autorestart: true,
    max_memory_restart: "1G",
    listen_timeout: 10000,
    max_restarts: 10,
    min_uptime: "5s",
    max_restart_delay: 3000
  }]
};
