const fs = require('fs');
function loadProtosFromFolder(folderPaths, protobuf) {
  const proto_files = [];
  console.log(folderPaths);
  for (folderPath of folderPaths) {
  console.log(folderPath);
  const files = fs.readdirSync(folderPath);
  files.forEach((file) => {
      if (file.endsWith('.proto')) {
        file = `${folderPath}/${file}`;
        proto_files.push(file);
      }
  });
  }
return protobuf.loadSync(proto_files);
}

module.exports = loadProtosFromFolder;