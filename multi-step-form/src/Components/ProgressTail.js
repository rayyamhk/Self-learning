import React from "react";

function ProgressTail({step, description, status}) {
    return(
        <div className={status==="finished" ? "progress-container finished" : status==="processing" ? "progress-container processing" : "progress-container not-reached"}>
            <div className="bar bar-left"></div>
            <div className="circle"></div>
            <h3>{"step " + step}</h3>
            <p>{description}</p>
        </div>
    );
}

export default ProgressTail;