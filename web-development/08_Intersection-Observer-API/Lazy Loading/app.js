const imgs = document.querySelectorAll(".lazy-loading");

const options = {
    root: null,
    //load the image before you actually "reach" the image
    rootMargin: "0px 0px 100px 0px",
    threshold: 0.1
}

const callback = imgs => {
    imgs.forEach(img => {
        if(img.isIntersecting) {
            img.target.src = img.target.dataset.url;
            observer.unobserve(img.target);
        }
    })
}

const observer = new IntersectionObserver(callback, options);

imgs.forEach(img => observer.observe(img));