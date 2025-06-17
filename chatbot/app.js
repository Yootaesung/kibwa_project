const express = require('express');
const { spawn } = require('child_process');
const path = require('path');
const app = express();

// 정적 파일 서빙
app.use(express.static(path.join(__dirname, 'templates')));
app.use(express.json());

// FastAPI 프록시
app.use('/api', (req, res) => {
  const python = spawn('python3', ['-m', 'uvicorn', 'app:app', '--host', '0.0.0.0', '--port', '8001']);
  
  let responseData = '';
  python.stdout.on('data', (data) => {
    responseData += data.toString();
  });

  python.stderr.on('data', (data) => {
    console.error(`stderr: ${data}`);
  });

  python.on('close', (code) => {
    if (code !== 0) {
      return res.status(500).json({ error: 'Internal Server Error' });
    }
    res.send(responseData);
  });
});

// 루트 경로
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'templates', 'test.html'));
});

const PORT = process.env.PORT || 8000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});