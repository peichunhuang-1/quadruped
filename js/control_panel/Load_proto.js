const fs = require('fs');
const path = require('path');
function loadProtosFromFolder(folderPaths, protobuf) {
  // const proto_files = {root: [], file: []};
  const proto_files = [];
  console.log(folderPaths);
  for (folderPath of folderPaths) {
  console.log(folderPath);
  const files = fs.readdirSync(folderPath);
  files.forEach((file) => {
      if (file.endsWith('.proto')) {
        // proto_files.root.push(folderPath);
        // proto_files.file.push(file);
        proto_files.push(folderPath + "/" + file);
      }
  });
  }
  console.log(proto_files);
return protobuf.loadSync(proto_files);
}

module.exports = loadProtosFromFolder;