const fs = require('fs');
function loadProtosFromFolder(folderPath, protobuf) {
const files = fs.readdirSync(folderPath);
const proto_files = [];
files.forEach((file) => {
    if (file.endsWith('.proto')) {
      file = `${folderPath}/${file}`;
      proto_files.push(file);
    }
});
return protobuf.loadSync(proto_files);
}

module.exports = loadProtosFromFolder;