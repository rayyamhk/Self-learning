import React from "react";

function Button({text, extra}) {
    return(
        <button type="button" className={`btn btn-${text}`} onClick={extra}>
            {text}
        </button>
    );
}

export default Button;