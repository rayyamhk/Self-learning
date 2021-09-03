const targets = document.querySelectorAll("section");
const navElements = document.querySelectorAll("ul li a");
const bubble = document.querySelector("#bubble");

//Animate Nav Bar
const option1 = {
    root: null,
    rootMargin: "0px",
    threshold: 0.5
}

const callback1 = entries => {
    entries.forEach(entry => {
        const navElement = navElements[parseInt(entry.target.dataset.index)];
        const rect = navElement.getBoundingClientRect();
        if(entry.isIntersecting) {
            navElement.style.color = "#f1c40f";
            bubble.style.height = rect.height + "px";
            bubble.style.width = rect.width + "px";
            bubble.style.top = rect.top + "px";
            bubble.style.left = rect.left + "px";    
        }
        else {
            navElement.style.color = "white";
        }
    })
}

let observer1 = new IntersectionObserver(callback1, option1);
targets.forEach(target => observer1.observe(target));


//Animate Paragraphs
const option2 = {
    root: null,
    rootMargin: "0px",
    threshold: 0.4
}

const callback2 = entries => {
    entries.forEach(entry => {
        const paras = document.querySelectorAll(`#${entry.target.id} p`);
        if(entry.isIntersecting) {
            for(let i = 0; i < paras.length; i++) {
                paras[i].style.opacity = "0";
                paras[i].style.transform = "translateX(-100px)";
                paras[i].style.animation = `text-in 1s ease-in-out ${paras[i].dataset.delay} forwards`;
            }
        }
        else {
            for(let i = 0; i < paras.length; i++) {
                paras[i].style.opacity = "1";
                paras[i].style.transform = "translateX(0)";
                paras[i].style.animation = `text-out 1s ease-in-out ${paras[i].dataset.delay} forwards`;
            }
        }
    })
}

let observer2 = new IntersectionObserver(callback2, option2);
targets.forEach(target => observer2.observe(target))


//Animate Headings
const option3 = {
    threshold: 0.2
}

const callback3 = entries => {
    entries.forEach(entry => {
        const heading = document.querySelector(`#${entry.target.id} h1`);
        if(entry.isIntersecting) {
            heading.style.opacity = "0";
            heading.style.transform = "translateY(-100px)";
            heading.style.animation = "heading-in 1s ease-in-out forwards";
        }
        else {
            heading.style.opacity = "1";
            heading.style.transform = "translateY(0)";
            heading.style.animation = "heading-out 1s ease-in-out forwards";
        }
    })
}

let observer3 = new IntersectionObserver(callback3, option3);
targets.forEach(target => observer3.observe(target));