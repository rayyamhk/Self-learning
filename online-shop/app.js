const shoppingCart = document.querySelector(".shopping-cart");
const shoppingList = document.querySelector(".shopping-cart-list");
const shoppingListBtn = document.getElementById("close-btn");
const shoppingListClrBtn = document.getElementById("clear-btn");

shoppingCart.addEventListener("click", function(e){
    shoppingList.style.width = "80%";
    document.querySelector(".overlay").style.visibility = "visible";
})

shoppingListBtn.addEventListener("click", function(e){
    shoppingList.style.width = "0%";
    document.querySelector(".overlay").style.visibility = "hidden";
})

shoppingListClrBtn.addEventListener("click", function(e){
    selectedProducts = [];
    repaint();
})

const boxes = document.querySelector(".boxes");
const goods = document.getElementById("goods");
const price = document.getElementById("current-price");
const products_num = document.getElementById("num-of-products")
let selectedProducts = [];

function DisplayProducts(product) {   
    const box = document.createElement("div");
    const imageContainer = document.createElement("div");
    const img = document.createElement("img");
    const tag = document.createElement("div");
    const i = document.createElement("i");
    const p1 = document.createElement("p");
    const p2 = document.createElement("p");

    box.classList.add("box");
    imageContainer.classList.add("product-image");
    img.src = product.fields.image;
    tag.classList.add("tag");
    i.classList.add("fas");
    i.classList.add("fa-cart-plus");
    p1.classList.add("product-name");
    p1.textContent = product.fields.title;
    p2.classList.add("product-price");
    p2.textContent = "$" + product.fields.price;

    box.appendChild(imageContainer);
    box.appendChild(p1);
    box.appendChild(p2);
    tag.appendChild(i);
    tag.appendChild(document.createTextNode("ADD TO CART"));
    imageContainer.appendChild(img);
    imageContainer.appendChild(tag);
    boxes.appendChild(box);

    tag.addEventListener("click", (e) => {
        if(!selectedBefore(product.fields.title)) {
            let temp = {
                name: product.fields.title,
                price: product.fields.price,
                amount: 1,
                image: product.fields.image
            }
            selectedProducts.push(temp);
        }
        else {
            updateSelectedProducts("up", product.fields.title);
        }
        repaint();
    })
}

function selectedBefore(name) {
    for(let i = 0; i < selectedProducts.length; i++) {
        if(selectedProducts[i].name === name) {
            return true;
        }
    }
    return false;
}

function updateSelectedProducts(task, name) {
    for(let i = 0; i < selectedProducts.length; i++) {
        if(selectedProducts[i].name === name) {
            if(task === "up") {
                selectedProducts[i].amount += 1;
            }
            if(task === "down") {
                if(selectedProducts[i].amount > 1) {
                    selectedProducts[i].amount -= 1;
                }
                else {
                    selectedProducts.splice(i, 1);
                }
            }
            if(task === "remove") {
                selectedProducts.splice(i, 1);
            }
        }
    }
}

function addShoppingCart(product) {
    const good = document.createElement("div");
    const img = document.createElement("img");
    const goodInfo = document.createElement("div");
    const goodName = document.createElement("span");
    const goodPrice = document.createElement("span");
    const remove = document.createElement("button");
    const goodControl = document.createElement("div");
    const up = document.createElement("i");
    const amount = document.createElement("span");
    const down = document.createElement("i");

    good.classList.add("good");
    img.src = product.image;
    goodInfo.classList.add("good-info");
    goodName.textContent = product.name;
    goodPrice.textContent = "$" + product.price;
    remove.classList.add("remove-btn");
    remove.textContent = "remove";
    goodControl.classList.add("good-control");
    up.classList.add("fas");
    up.classList.add("fa-sort-up");
    up.classList.add("arrow-up");
    down.classList.add("fas");
    down.classList.add("fa-sort-down");
    down.classList.add("arrow-down");
    amount.textContent = product.amount;

    good.appendChild(img);
    good.appendChild(goodInfo);
    good.appendChild(goodControl);
    goodInfo.appendChild(goodName);
    goodInfo.appendChild(goodPrice);
    goodInfo.appendChild(remove);
    goodControl.appendChild(up);
    goodControl.appendChild(amount);
    goodControl.appendChild(down);
    goods.appendChild(good);

    remove.addEventListener("click", (e) => {
        updateSelectedProducts("remove", product.name);
        repaint();
    })
    up.addEventListener("click", (e) => {
        updateSelectedProducts("up", product.name);
        repaint();
    })
    down.addEventListener("click", (e) => {
        updateSelectedProducts("down", product.name);
        repaint();
    })
}

function repaint() {
    goods.textContent = "";
    selectedProducts.forEach(product => addShoppingCart(product));
    let totalPrice = 0;
    selectedProducts.forEach(product => {
        let temp = product.price * product.amount;
        totalPrice += temp;
    })
    price.textContent = totalPrice;
    products_num.textContent = selectedProducts.length;
}

const products = [
    {
        "sys": { "id": "1" },
        "fields": {
            "title": "queen panel bed",
            "price": 10.99,
            "image": "./img/product-1.jpeg"
        }
    },
    {
        "sys": { "id": "2" },
        "fields": {
            "title": "king panel bed",
            "price": 12.99,
            "image": "./img/product-2.jpeg"
        }
    },
    {
        "sys": { "id": "3" },
        "fields": {
            "title": "single panel bed",
            "price": 12.99,
            "image": "./img/product-3.jpeg"
        }
    },
    {
        "sys": { "id": "4" },
        "fields": {
            "title": "twin panel bed",
            "price": 22.99,
            "image": "./img/product-4.jpeg"
        }
    },
    {
        "sys": { "id": "5" },
        "fields": {
            "title": "fridge",
            "price": 88.99,
            "image": "./img/product-5.jpeg"
        }
    },
    {
        "sys": { "id": "6" },
        "fields": {
            "title": "dresser",
            "price": 32.99,
            "image": "./img/product-6.jpeg"
        }
    },
    {
        "sys": { "id": "7" },
        "fields": {
            "title": "couch",
            "price": 45.99,
            "image": "./img/product-7.jpeg"
        }
    },
    {
        "sys": { "id": "8" },
        "fields": {
            "title": "table",
            "price": 33.99,
            "image": "./img/product-8.jpeg"
        }
    }
]

products.forEach(product => DisplayProducts(product));
