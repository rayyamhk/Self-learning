import React from 'react';

function Likes({likes, showLike}) {
    return(
        <div className={showLike?"like-tag":"like-tag like-tag-hide"}>
            <i className="fas fa-heart"></i>
            <span>{likes}</span>
        </div>
    )
}

export default Likes;