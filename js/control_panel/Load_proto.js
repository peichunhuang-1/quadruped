const fs = require('fs');
const path = require('path');
function loadProtosFromFolder(folderPaths, protobuf) {
  const proto_files = {root: [], file: []};
  console.log(folderPaths);
  for (folderPath of folderPaths) {
  console.log(folderPath);
  const files = fs.readdirSync(folderPath);
  files.forEach((file) => {
      if (file.endsWith('.proto')) {
        proto_files.root.push(folderPath);
        proto_files.file.push(file);
      }
  });
  }
return protobuf.loadSync(proto_files);
}

module.exports = loadProtosFromFolder;