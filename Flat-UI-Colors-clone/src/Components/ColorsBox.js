import React from "react";
import {Link} from "react-router-dom";

function ColorsBox({colors}) {
    return(
        <Link to={"/palette/" + colors.id} className="colors-box-anchor">
            <div className="colors-box">
                <div>
                    {
                        colors.colorSet.map((colorInfo, index) => <span style={{backgroundColor: colorInfo.hex1, minHeight: "40px"}} key={index}></span>)
                    }
                </div>  
                <span>{colors.title}</span>
            </div>
        </Link>
    )
}

export default ColorsBox;