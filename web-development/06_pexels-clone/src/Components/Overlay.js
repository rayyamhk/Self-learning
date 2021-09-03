import React from 'react';

function Overlay({image, isOpen, toggle}) {
    return(
        <div className={isOpen ? "overlay-wrapper-open" : "overlay-wrapper-closed"} onClick={() => toggle()}>
            <div className="overlay-img-container">
                <img src={image.url} alt=""></img>
            </div>
        </div>
    )
}
export default Overlay;