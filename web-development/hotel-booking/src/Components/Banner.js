import React from "react";

// Everything inside the Banner tag will be regarded as children and rendered automatically
function Banner({children, title, description}) {
    return(
        <div className="banner">
            <h1>{title}</h1>
            <div></div>
            <p>{description}</p>
            {/* children refers to button */}
            {children} 
        </div>
    )
}

export default Banner;