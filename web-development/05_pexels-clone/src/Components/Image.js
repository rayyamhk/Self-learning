import React from 'react';
import Likes from "./Likes";

function Image({img, showLike, imgToggle, likeShow, likeHide}) {
    return(
        <div className="img-container" onClick={() => imgToggle(img)}>
            <img src={img.url} alt=""></img>
            <div className="img-info">
                <span>{img.author}</span>
                <div className="wrapper">
                    <i className="far fa-plus-square"></i>
                    <i className="far fa-heart" onMouseEnter={() => likeShow()} onMouseLeave={() => likeHide()}>
                        <Likes likes={img.likes} showLike={showLike}/>
                    </i>
                </div>
            </div>
        </div>
    );
}

export default Image;