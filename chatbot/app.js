module.exports = {
  apps: [{
    name: 'kibwa-chatbot',
    script: 'app.js',
    interpreter: 'node',
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '1G',
    env: {
      NODE_ENV: 'production',
      PATH: process.env.PATH,
      VIRTUAL_ENV: '/work/venv',
      PYTHONUNBUFFERED: '1'
    },
    env_production: {
      NODE_ENV: 'production',
      PATH: process.env.PATH,
      VIRTUAL_ENV: '/work/venv',
      PYTHONUNBUFFERED: '1'
    },
    cwd: '/work/kibwa_project/chatbot',
    log_date_format: 'YYYY-MM-DD HH:mm:ss',
    error_file: 'logs/error.log',
    out_file: 'logs/out.log',
    merge_logs: true,
    exec_mode: 'fork',
    time: true
  }]
};
