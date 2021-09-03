const targets = document.querySelectorAll("section");
const navElements = document.querySelectorAll("ul li a");
const bubble = document.querySelector("#bubble");

const options = {
    root: null, // viewport as default, i.e. document.querySelector("window");
    rootMargin: "0px",
    threshold: 0.5 
    //if the viewport displays more than 50% of the target,
    //it will be treated as "Is Intersecting", otherwise it will be treated as "Not Intersecting"
}

//The entries is an array. During the initialization, the entries contains all targets
//After that, the entries only contains visible targets
const callback = entries => {
    entries.forEach(entry => {
        const navElement = navElements[parseInt(entry.target.dataset.index)];
        const rect = navElement.getBoundingClientRect();
        const paras = document.querySelectorAll(`#${entry.target.id} p`);
        //For the target we are entering
        if(entry.isIntersecting) {
            navElement.style.color = "#f1c40f";
            bubble.style.height = rect.height + "px";
            bubble.style.width = rect.width + "px";
            bubble.style.top = rect.top + "px";
            bubble.style.left = rect.left + "px";

            for(let i = 0; i < paras.length; i++) {
                paras[i].style.opacity = "0";
                paras[i].style.transform = "translateX(-100px)";
                paras[i].style.animation = `text-in 1s ease-in-out ${paras[i].dataset.delay} forwards`;
            }
        }
        //For the target we are leaving
        else {
            navElement.style.color = "white";

            for(let i = 0; i < paras.length; i++) {
                paras[i].style.opacity = "1";
                paras[i].style.transform = "translateX(0)";
                paras[i].style.animation = `text-out 1s ease-in-out ${paras[i].dataset.delay} forwards`;
            }
        }
    })
}

//Define an observer to observe all targets
let observer = new IntersectionObserver(callback, options);

//"Register" the targets
targets.forEach(target => observer.observe(target));