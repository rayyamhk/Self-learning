import React from "react";

function Overlay({color, colorFormat}) {
    const divStyle = {
        backgroundColor: ""
    }
    if(colorFormat === "hex2") {
        divStyle.backgroundColor = "#" + color;
    }
    else {
        divStyle.backgroundColor = color;
    }
    return(
        <div className="overlay" style={divStyle}>
            <div>
                <span className="overlay-popup">copied</span>
                <span className="overlay-color">{color}</span>
            </div>
        </div>
    )
}

export default  Overlay;