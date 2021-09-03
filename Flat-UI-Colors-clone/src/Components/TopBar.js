import React, {useState} from "react";
import {Link} from "react-router-dom";

function TopBar({sound, selectColorFormat, soundControl}) {
    const [isClick, setClick] =  useState(false);
    return(
        <div className="top-bar">
            <Link to="/" className="go-back">back</Link>
            <div onClick={() => {
                isClick?setClick(false):setClick(true);
            }}>
                <span>copy format: </span>
                <span className="copy-format">hex1 (#aa1923)</span>
                <div className={isClick?"format-list":"format-list-closed"}>
                    <span onClick={(e)=>selectColorFormat(e)}>hex1 (#aa1923)</span>
                    <span onClick={(e)=>selectColorFormat(e)}>hex2 (aa1923)</span>
                    <span onClick={(e)=>selectColorFormat(e)}>rgb - (1,2,3)</span>
                    <span onClick={(e)=>selectColorFormat(e)}>rgba - (1,2,3,0.4)</span>
                </div>
            </div>
            <span className="sound-control" onClick={() => soundControl()}>{sound?"sound on":"sound off"}</span>
        </div>
    )
}

export default TopBar;