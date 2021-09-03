import React from "react";
import ColorsBox from "../Components/ColorsBox";
import ColorSet from "../Data/ColorSet";
import SEO from "../Components/SEO";

function LandingPage() {
    return(
        <div className="container">
            <SEO title={"Palettes | Flat UI Colors"}/>
            {
                ColorSet.map((colors, index) => <ColorsBox colors={colors} key={index}/>)
                
            }
        </div>
    )
}

export default LandingPage;