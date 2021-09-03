const container = document.querySelector(".photos-container");
const footer = document.querySelector("footer");
const img = document.querySelector("img");

const options = {
    root: null,
    rootMargin: "0px",
    threshold: 0.1
}

//whenever you reach the footer, it loads 10 pictures
//it prevents the "observe & unobserve loop"
const callback = entries => {
    entries.forEach(entry => {
        if(entry.isIntersecting) {
            loadPics();
        }
    })
}

const observer = new IntersectionObserver(callback, options);
observer.observe(footer);

const loadPics = () => {
    for(let i = 0; i < 10; i++) {
        const img = new Image() || document.createElement("img");
        img.src = "https://picsum.photos/300";
        container.appendChild(img);
    }
}

//observe-unobserve loop
//change the initial observe to the image

// const callback = entries => {
//     entries.forEach(entry => {
//         if(entry.isIntersecting) {
//             const img = new Image() || document.createElement("img");
//             img.src = "https://picsum.photos/300";
//             container.appendChild(img);
//             observer.observe(img);
//             observer.unobserve(entry.target);
//         }
//     })
// }