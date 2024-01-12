let ip_addr = window.prompt('WebServer Address: [IP : port]');
let event_socket;
if (ip_addr) {
    event_socket = new WebSocket('ws://'+ip_addr);
}
let locked = true;
let input = {
    type : 0,
    zeta : 0,
    stanceheight : 0.1,
    liftheight : 0,
    steplength : 0,
    digital: false, 
    signal: false,
    power: false,
    vicon_trigger: false,
    orin_trigger: false,
    mode: 0
};

const send_cb = function() {
    if (event_socket) {
        event_socket.send(JSON.stringify(input));
    }
}

function range_slider_group(json) {
    const ranges = document.querySelectorAll('input[type=range]');
    for (var i = 0; i < ranges.length; i++) {
        ranges[i].addEventListener("input", function() {
        const label = this.parentElement.querySelector(".selected");
        label.innerHTML = this.value;
        });
    };

    for (var i = 0; i < ranges.length; i++) {
        ranges[i].addEventListener("change", function() {
        json[this.parentElement.id] = this.value / 1000.0;
        });
    };
    const selectors = document.querySelectorAll('select');
    for (var i = 0; i < selectors.length; i++) {
        selectors[i].addEventListener("change", function() {
        json[this.parentElement.id] = this.value;
        });
    }
    const buttons = document.querySelectorAll('.button');
    for (var i = 0; i < buttons.length; i++) {
        buttons[i].addEventListener("click", function() {
        this.classList.toggle('active');
        json[this.id] = this.classList.contains('active');
        send_cb();
        });
    }
}

function locker_cb() {
    var lockImage = document.getElementById('locker_img');
    var currentSrc = lockImage.getAttribute('src');
    if (currentSrc === './svg/lock.svg') {
        locked = false;
        lockImage.setAttribute('src', './svg/unlock.svg');
    } else {
        locked = true;
        lockImage.setAttribute('src', './svg/lock.svg');
    }
}

range_slider_group(input);



setInterval(()=>{
    if (!locked) {
        if (event_socket) {
            event_socket.send(JSON.stringify(input));
        }
    }
}, 500);