import React from "react";

const Copy = str => {
    const el = document.createElement("textarea");
    el.value = str;
    el.setAttribute('readonly', '');                // Make it readonly to be tamper-proof
    el.style.position = 'absolute';                 
    el.style.left = '-9999px';                      // Move outside the screen to make it invisible
    document.body.appendChild(el);                  // Append the <textarea> element to the HTML document
    const selected =            
        document.getSelection().rangeCount > 0        // Check if there is any content selected previously
        ? document.getSelection().getRangeAt(0)     // Store selection if found
        : false;                                    // Mark as false to know no selection existed before
    el.select();                                    // Select the <textarea> content
    document.execCommand('copy');                   // Copy - only works as a result of a user action (e.g. click events)
    document.body.removeChild(el);                  // Remove the <textarea> element
    if (selected) {                                 // If a selection existed before copying
        document.getSelection().removeAllRanges();    // Unselect everything on the HTML document
        document.getSelection().addRange(selected);   // Restore the original selection
    }
}

function ColorElement({color, colorFormat, text, selectColor}) {
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
        <div className="color-element" style={divStyle} onClick={() => {
            Copy(color);
            selectColor(color);
            document.querySelector(".overlay").style.zIndex = "1";
            document.querySelector(".overlay").style.opacity = "1";
            setTimeout(() => {
                document.querySelector(".overlay").style.zIndex = "-1";
                document.querySelector(".overlay").style.opacity = "0";
            }, 750);

        }}>
            <button>copy</button>
            <span>{text}</span>
        </div>
    )
}

export default ColorElement;