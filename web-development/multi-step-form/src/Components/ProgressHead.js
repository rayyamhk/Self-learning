import React from "react";

function ProgressHead({step, description, status}) {
    return(
        <div className={status==="finished" ? "progress-container finished" : status==="processing" ? "progress-container processing" : "progress-container not-reached"}>
            <div className="circle"></div>
            <div className="bar bar-right"></div>
            <h3>{"step " + step}</h3>
            <p>{description}</p>
        </div>
    );
}

export default ProgressHead;