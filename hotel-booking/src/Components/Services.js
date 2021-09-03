import React from "react";

import {FaCocktail, FaHiking, FaShuttleVan, FaBeer} from "react-icons/fa";
// import FaHiking from "react-icons/fa";
// import FaShuttleVan from "react-icons/fa";
// import FaBeer from "react-icons/fa"

function Services({children}) {
    return(
        <div className="services">
            {children}
            <div className="services-center">
                <div className="service">
                    <span><FaCocktail/></span>
                    <h6>Free Cocktails</h6>
                    <p>Lorem ipsum dolor sit amet consectetur adipisicing elit. Magni, corporis!</p>
                </div>
                <div className="service">
                    <span><FaHiking/></span>
                    <h6>Endless Hiking</h6>
                    <p>Lorem ipsum dolor sit amet consectetur adipisicing elit. Magni, corporis!</p>
                </div>
                <div className="service">
                    <span><FaShuttleVan/></span>
                    <h6>Free Shuttle</h6>
                    <p>Lorem ipsum dolor sit amet consectetur adipisicing elit. Magni, corporis!</p>
                </div>
                <div className="service">
                    <span><FaBeer/></span>
                    <h6>Strongest Beer</h6>
                    <p>Lorem ipsum dolor sit amet consectetur adipisicing elit. Magni, corporis!</p>
                </div>
            </div>

        </div>
    )
}

export default Services;