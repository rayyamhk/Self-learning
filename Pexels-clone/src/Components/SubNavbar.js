import React from 'react';

function SubNavbar({path, subnavToggle}) {
    return(
        <div className="subnav-container">
            <div name="photos" className={path==="photos"?"here":""} onClick={() => subnavToggle("photos") }>
                <i className="fas fa-image"></i>
                <span>Photos</span>
                <div className="underline"></div>
            </div>
            <div name="videos" className={path==="videos"?"here":""} onClick={() => subnavToggle("videos") }>
                <i className="fas fa-video"></i>
                <span>Videos</span>
                <div className="underline"></div>
            </div>
            
        </div>
    )
}
export default SubNavbar