import React from "react";

// Everything inside the Hero tag will be regarded as children and rendered automatically
function Hero({children, name, backgroundImg}) {
    return(
        <div className={name} style={backgroundImg &&  {backgroundImage: `url(${backgroundImg})`}}>
            {/* children refers to banner */}
            {children}
        </div>
    );
}

export default Hero;