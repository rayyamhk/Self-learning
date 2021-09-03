import React from 'react';
import Image from "./Image";
import images from "../images";
import videos from "../videos";

function ImageShowcase({path, width, showLike, imgToggle, likeShow, likeHide}) {
    let items1 = [], items2 = [], items3 = [], items4 = [];
    let temp = [];
    if(path === "photos") {
        temp = images;
    }
    if(path === "videos") {
        temp = videos;
    }
    for(let i = 0; i < temp.length; i++) {
        let img = temp[i];
        if(i%4===0) {
            items1.push(<Image key={img.id} img={img} showLike={showLike} imgToggle={imgToggle} likeShow={likeShow} likeHide={likeHide}/>);
        }
        else if(i%4===1) {
            items2.push(<Image key={img.id} img={img} showLike={showLike} imgToggle={imgToggle} likeShow={likeShow} likeHide={likeHide}/>);
        }
        else if(i%4===2) {
            items3.push(<Image key={img.id} img={img} showLike={showLike} imgToggle={imgToggle} likeShow={likeShow} likeHide={likeHide}/>);
        }
        else {
            items4.push(<Image key={img.id} img={img} showLike={showLike} imgToggle={imgToggle} likeShow={likeShow} likeHide={likeHide}/>);
        }
    }
    return (
        <div className="image-showcase">
            {
                width <= 768 ? (
                    <div className="img-wrapper">
                        {items1}
                        {items2}
                        {items3}
                        {items4}
                    </div>    
                ) : width > 768 && width <=1200 ? (
                    <>
                    <div className="img-wrapper">
                        {items1}
                        {items2}
                    </div> 
                    <div className="img-wrapper">
                        {items3}
                        {items4}
                    </div> 
                    </>
                ) : (
                    <>
                    <div className="img-wrapper">
                        {items1}
                    </div>
                    <div className="img-wrapper">
                        {items2}
                    </div>  
                    <div className="img-wrapper">
                        {items3}
                    </div> 
                    <div className="img-wrapper">
                        {items4}
                    </div> 
                    </>
                )
            }
        </div>
    )
}

export default ImageShowcase;