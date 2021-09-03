const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const x_correct = canvas.getBoundingClientRect().left;
const y_correct = canvas.getBoundingClientRect().top;
let gameRef;
let timeRef;
let time = 0;
let isStart = false;

const progressBar = document.querySelector(".rows-wrapper");
const level = document.querySelector(".level span");
const minute = document.querySelector(".min");
const second = document.querySelector(".sec");
const restart = document.querySelector(".gameover-panel button");
const gameoverPanel = document.querySelector(".gameover-panel");

const ghost = document.getElementById("ghost");
const you = document.getElementById("you");
let enemies = [];

const levelUpSound = document.getElementById("level");
levelUpSound.volume = 0.25;
const bgMusic = document.getElementById("bg");
bgMusic.volume = 0.5;
const death = document.getElementById("death");
death.volume = 0.5;


const You = {
    x: canvas.width * 0.5,
    y: canvas.height * 0.5,
    size: 30,
    draw: function() {
        ctx.drawImage(you, this.x, this.y, this.size, this.size);
    }
}

function initialization(num) {
    timeConvert(0);
    level.textContent = "1";
    for(let i = 0; i < num; i++) {
        let size = Math.floor(Math.random() * 20) + 15;
        let x = Math.floor(Math.random()*(canvas.width-size));
        let y = Math.floor(Math.random()*(canvas.height-size));
        let dx = Math.floor(Math.random()*2) + 1.5;
        let dy = Math.floor(Math.random()*2) + 1.5;
        if(Math.floor(Math.random()*9)%2===0) {
            dx *= -1;
        }
        if(Math.floor(Math.random()*9)%2===0) {
            dy *= -1;
        }
        generateBall(x, y, dx, dy, size);
    }
    enemies.forEach(enemy => enemy.draw());
    canvas.addEventListener("mousemove", repaint);
}

function start() {
    isStart = true;
    bgMusic.play();
    progressBar.style.animationName = "stretch";
    timeRef = setInterval(() => {
        time += 1;
        timeConvert(time);
        if(time % 5 === 0) {
            nextLevel();
        }
    }, 1000)
    repaint();
}

function end() {
    isStart = false;
    bgMusic.pause();
    bgMusic.currentTime = 0;
    death.play();
    clearInterval(timeRef);
    time = 0;
    progressBar.style.animationName = "";
    window.cancelAnimationFrame(gameRef);
    enemies = [];
    repaint();
    canvas.style.display = "none";
    gameoverPanel.style.display = "flex";
}

function nextLevel() {
    level.textContent = parseInt(level.textContent) + 1;
    levelUpSound.play();
    enemies.forEach(enemy => {
        if(Math.abs(enemy.dx) < 10 && Math.abs(enemy.dy) < 10) {
            if(time <= 30) {
                enemy.dx *= 1.2;
                enemy.dy *= 1.2;
            }
            else {
                enemy.dx *= 1.1;
                enemy.dy *= 1.1;
            }
        }
    })
}

function timeConvert(time) {
    let min = Math.floor(time/60);
    let sec = time % 60;
    if(sec < 10) {
        sec = "0" + sec;
    } 
    if(min < 10) {
        min = "0" + min;
    }
    minute.textContent = min;
    second.textContent = sec;
}

function generateBall(x, y, dx, dy, size) {
    let temp = {
        x: x,
        y: y,
        dx: dx,
        dy: dy,
        size: size,
        draw: function() {
            if(this.x + this.size >= canvas.width || this.x <= 0) {
                this.dx *= -1;
            }
            if(this.y + this.size >= canvas.height || this.y <= 0) {
                this.dy *= -1;
            }
            ctx.drawImage(ghost, this.x, this.y, this.size, this.size);
        }
    }
    enemies.push(temp);
}

function repaint(e) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    You.draw();
    if(isStart) {
        enemies.forEach(enemy => {
            enemy.x += enemy.dx;
            enemy.y += enemy.dy;
            enemy.draw();
        });
        if(!checkBoundaries() || !checkCollision()) {
            end();
        }
        else {
            gameRef = window.requestAnimationFrame(repaint);
        }
    }
    else {
        enemies.forEach(enemy => enemy.draw());
    }
}

function checkBoundaries() {
    if(You.x + You.size >= canvas.width || You.x <= 4) {
        return false;
    }
    if(You.y + You.size >= canvas.height || You.y <= 4) {
        return false;
    }
    return true;
}

function checkCollision() {
    for(let i = 0; i < enemies.length; i++) {
        if(hasCollision(You, enemies[i])) {
            return false;
        }
    }
    return true;
}

function hasCollision(main, enemy) {
    let x1 = main.x, y1 = main.y, r1 = main.size;
    let x2 = enemy.x, y2 = enemy.y, r2 = enemy.size;
    let x_min = x1 - r2, x_max = x1 + r1;
    let y_min = y1 - r2, y_max = y1 + r1;
    if(x2 >= x_min && x2 <= x_max && y2 >= y_min && y2 <= y_max) {
        return true;
    }
    else{
         return false;
    }
}

canvas.addEventListener("click", (e) => {
    if(!isStart && enemies.length > 0) {
        canvas.removeEventListener("mousemove", repaint);
        start();
    }
});

canvas.addEventListener("mousemove", (e) => {
    You.x = e.clientX - x_correct;
    You.y = e.clientY - y_correct;
})

restart.addEventListener("click", (e) => {
    gameoverPanel.style.display = "none";
    canvas.style.display = "initial";
    death.pause();
    death.currentTime = 0;
    initialization(8);
})

window.addEventListener("load", (e) => {
    initialization(8);
    window.alert("1. Please enable autoplay in your broswer, otherwise some sound effects will be blocked.\n 2. Find a good position for your star.\n 3. Once you click, the game starts and try your best to survive.")
})