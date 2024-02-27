const protobuf = require('protobufjs');
const ServiceClient = require('./../core/ServiceClient.js');
const http = require('http');
const fs = require('fs');
const WebSocket = require('ws');
const path = require('path');  
const proto_loader = require('./Load_proto.js');  
const proto_Path = `${process.env.PROTO_PATH}`;
const root = proto_loader(["./../../protos", proto_Path], protobuf);
console.log(root);
function reply_call_back(reply) {
    console.log(reply);
}
const gait_cmd_clt = ServiceClient("gait/command", reply_call_back, root, 'robot_msg.GaitRequest', 'robot_msg.GaitReply');
const power_cmd_clt = ServiceClient("power/command", reply_call_back, root, 'power_msg.PowerBoardStamped', 'power_msg.PowerBoardStamped');

const server = http.createServer(function(request, response) {
    let filePath = '.' + request.url;
    const allowedDirectories = ['./', './svg', './css', './js', './html'];
    const isValidDirectory = allowedDirectories.some(dir => filePath.startsWith(dir));
    if (!isValidDirectory) {
      response.writeHead(403);
      response.end('Access Forbidden');
      return;
    }
    if (filePath === './') {
      filePath = './index.html';
    }
  
    const extname = path.extname(filePath);
    let contentType = 'text/html';
    switch (extname) {
        case '.js':
          contentType = 'text/javascript';
          break;
        case '.css':
          contentType = 'text/css';
          break;
        case '.svg':
          contentType = 'image/svg+xml';
          break;
        default:
          contentType = 'text/html';
      }
    
      fs.readFile(filePath, (err, content) => {
        if (err) {
          if (err.code === 'ENOENT') {
            response.writeHead(404);
            response.end('File not found');
          } else {
            response.writeHead(500);
            response.end(`Server Error: ${err.code}`);
          }
        } else {
          response.writeHead(200, { 'Content-Type': contentType });
          response.end(content, 'utf-8');
        }
      });
});

const port = 8080;
server.listen(port, '0.0.0.0', function() {
  console.log(`Server is running on port ${port}`);
});

const wss = new WebSocket.Server({ server });
wss.on('connection', function(ws) {
console.log('Client connected');
ws.on('message', function(message) {
    const data = JSON.parse(message);
    gait_cmd_clt.send({
      type : data.type,
      zeta : data.zeta,
      stanceheight : data.stanceheight,
      liftheight : data.liftheight,
      steplength : data.steplength
    });
    
    power_cmd_clt.send({
      digital: {'digital': data.digital, 'signal': data.signal, 'power': data.power, 'vicon_trigger': data.vicon_trigger, 'orin_trigger': data.orin_trigger, 'clean_error': true}, 
      analog: {},
      mode: data.mode
    })
    console.log(data);
});

ws.on('close', function() {
    console.log('Client disconnected');
});
});