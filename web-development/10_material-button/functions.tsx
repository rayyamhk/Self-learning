
export const getGeometries = (e: React.MouseEvent, element?: HTMLElement): number[] => {
    let target: HTMLElement;
    if(element) {
        target = element;
    }
    else {
        target = e.target as HTMLElement;
    }
    const { left, top, width, height } = target.getBoundingClientRect();
    const cordX = e.nativeEvent.clientX - left;
    const cordY = e.nativeEvent.clientY - top;
    return [cordX, cordY, width, height];
}

export const RippleTrigger = (e: React.MouseEvent, ripple: HTMLElement) => {
    const button = e.currentTarget as HTMLButtonElement;
    if(ripple) {
        const [cordX, cordY, width, height] = getGeometries(e, button);
        const radius = Math.max(width, height);
        ripple.style.left = cordX - radius + "px";
        ripple.style.top = cordY - radius + "px";
        ripple.style.width = radius * 2 + "px";
        ripple.style.height = radius * 2 + "px";
        ripple.classList.remove("ripple");
        void ripple.offsetWidth;
        ripple.classList.add("ripple");
    }
}

export const isHex = (hex: string) => {
    const regexp = /[0-9a-fA-F]/g;
    if(hex.length === 6) {
        const red = hex.substring(0,2);
        const green = hex.substring(2,4);
        const blue = hex.substring(4,6);
        const r = red.match(regexp) || [];
        const g = green.match(regexp) || [];
        const b = blue.match(regexp) || [];
        if(r.length === 2 && g.length === 2 && b.length === 2) {
            return true;
        }
        else {
            return false;
        }
    }
    else {
        return false;
    }
}

export const BoxShadowStyle = (hover: boolean, color: string) => {
    let rgb = color
    const hashtag = rgb[0];
    if(hashtag === "#") {
        rgb = color.substring(1);
    }
    if(!hover) {
        return `0px 2px 2px 0px ${getRGBA(rgb, 0.2)}, 0px 3px 1px -2px ${getRGBA(rgb, 0.25)}, 0px 1px 5px 0px ${getRGBA(rgb, 0.15)}`;
    }
    else {
        return `0px 0px 25px 2px ${getRGBA(rgb, 0.2)}, 0px 15px 15px -10px ${getRGBA(rgb, 0.7)}, 0px 1px 5px 0px ${getRGBA(rgb, 0.9)}`;
    }
}

export const getBrandColor = (brand: string) => {
    let backgroundColor;
    let BrandTag = "Fa" + brand.charAt(0).toUpperCase() + brand.substring(1);
    switch(brand) {
        case "twitter":
            backgroundColor = "#55ACEF";
            break;
        case "facebook":
            backgroundColor = "#3B5999";
            BrandTag += "Square";
            break;
        case "instagram":
            backgroundColor = "#DD3066";
            break;
        case "google":
            backgroundColor = "#DD4C39";
            break;
        case "linkedin":
            backgroundColor = "#0876B3";
            break;
        case "youtube":
            backgroundColor = "#E42D27";
            break;
        case "tumblr":
            backgroundColor = "#35455C";
            break;
        case "github":
            backgroundColor = "#333333";
            break;
        case "reddit":
            backgroundColor = "#FF4400";
            break;
        case "whatsapp":
            backgroundColor = "#40e15d";
            break;
        case "line":
            backgroundColor = "#3fc004";
            break;
        case "telegram":
            backgroundColor = "#35a9dc";
            BrandTag += "Plane";
            break;
        default:
            backgroundColor = "#FFFFFF";
            break;
    }
    return { backgroundColor, BrandTag };
}

export const getRGBA = (color: string, opacity: number = 1) => {   
    //valid input: #ffffff or ffffff 
    if(opacity < 0 || opacity > 1) {
        opacity = 1;
    }
    const hashtag = color[0];
    let hex = color;
    if(hashtag === "#") {
        hex = color.substring(1);
    }
    if(isHex(hex)) {
        const red = hex.substring(0,2);
        const green = hex.substring(2,4);
        const blue = hex.substring(4,6);
        return `rgba(${parseInt("0x"+red)}, ${parseInt("0x"+green)}, ${parseInt("0x"+blue)}, ${opacity})`;
    }
    else {
        throw Error("Invalide Input");
    }
}

export const halfTheSize = (size: string) => {
    const digit = /[0-9.]/g;
    const letter = /[a-zA-Z]/g;
    const digitResult = size.match(digit);
    const unitResult = size.match(letter);
    if(digitResult && unitResult) {
        return (parseFloat(digitResult.join("")) / 2).toString() + unitResult.join("");
    }
    else {
        throw Error("Invalid Input.");
    }  
}