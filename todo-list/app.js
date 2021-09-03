const todoList = document.querySelector(".todo-list");
const addBtn = document.querySelector(".add-btn");
const inputArea = document.querySelector("input[name=todo]");
const main = document.querySelector("main");

let listIsClick = false;
let currentList = "";

start();

addBtn.addEventListener("click", (e) => {
    const content = inputArea.value;
    if(inputArea.value !== "") {
        createTODO(content);
        appendLocalStorage(content);
    }
})

function createTODO(content) {
    const list = document.createElement("li");
    const listContent = document.createElement("span");
    const button = document.createElement("button");
    const logo = document.createElement("i");
    const tickContainer = document.createElement("div");
    const tick = document.createElement("div");

    listContent.textContent = content;
    button.classList.add("cancel-btn");
    logo.classList.add("fas");
    logo.classList.add("fa-times");
    tickContainer.classList.add("tick-icon");
    listContent.contentEditable = true;

    tickContainer.appendChild(tick);
    button.appendChild(logo);
    list.appendChild(tickContainer);
    list.appendChild(listContent);
    list.appendChild(button);
    todoList.appendChild(list);   

    listContent.addEventListener("input", (e) => {
        replaceLocalStorage(currentList, listContent.textContent);
        currentList = listContent.textContent;
    })
    button.addEventListener("click", (e) => {
        todoList.removeChild(list);
        let temp = listContent.textContent;
        removeLocalStorage(temp);
    })
    listContent.addEventListener("click", (e) => {
        currentList = listContent.textContent;
        if(listIsClick) {
            list.classList.remove("list-done");
            listIsClick = false;
        }
        else {
            list.classList.add("list-done");
            listIsClick = true;
        }
    })
    
    inputArea.value = "";
    main.scroll(0, main.scrollHeight);
}

function appendLocalStorage(content) { 
    let temp = JSON.parse(localStorage.getItem("todo-list"));
    temp.push(content);
    console.log(JSON.stringify(temp));
    localStorage.setItem("todo-list", JSON.stringify(temp));
}

function removeLocalStorage(content) {
    let temp = JSON.parse(localStorage.getItem("todo-list"));
    for(let i = 0; i < temp.length; i++) {
        if(temp[i] === content) {
            temp.splice(i,1);
            break;
        }
    }
    localStorage.setItem("todo-list", JSON.stringify(temp));
}

function replaceLocalStorage(before, after) {
    let temp = JSON.parse(localStorage.getItem("todo-list"));
    for(let i = 0; i < temp.length; i++) {
        if(temp[i] === before) {
            temp[i] = after;
            break;
        }
    }
    localStorage.setItem("todo-list", JSON.stringify(temp));
}

function start() {
    if(!localStorage.getItem("todo-list")) {
        let temp = [];
        localStorage.setItem("todo-list", JSON.stringify(temp));
    } else {
        displayLists();
    }
}

//Note that localstorage only accept string
function displayLists() {
    let lists = JSON.parse(localStorage.getItem("todo-list"));
    lists.forEach(list => {
        console.log(list);
        createTODO(list);
    });
}