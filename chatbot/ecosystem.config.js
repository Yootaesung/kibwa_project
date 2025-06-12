module.exports = {
  apps: [{
    name: "fastapi-chatbot",
    script: "/work/kibwa_project/chatbot/run.sh",
    interpreter: "bash",
    cwd: "/work/kibwa_project/chatbot",
    watch: false,
    env: {
      NODE_ENV: "production",
      PYTHONUNBUFFERED: "1"
    },
    error_file: "./logs/error.log",
    out_file: "./logs/out.log",
    log_date_format: "YYYY-MM-DD HH:mm:ss",
    merge_logs: true,
    time: true,
    autorestart: true,
    max_memory_restart: "1G",
    listen_timeout: 10000,
    max_restarts: 10,
    max_memory_restart: "1G",
    network_mode: "host"
  }]
}
