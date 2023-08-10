//setupProxy.js
const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function(app) {

  //app.use는 미들웨어를 사용할때 사용
  // '/api'로  요청이 오면, proxy미들웨어를 실행
  app.use(
    createProxyMiddleware('/api' , {
      //spring boot 포트는 8080번이므로 target을 8080으로 설정 
      target: 'http://localhost:8080',
      changeOrigin: true,
    })
  );
};