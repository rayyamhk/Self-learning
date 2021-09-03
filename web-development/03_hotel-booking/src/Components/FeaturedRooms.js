import React from "react";
import Room from "./Room";

function FeaturedRooms({children}) {
    return(
        <div className="featured-rooms">
            {children}
            <div className="featured-rooms-center">
                <Room name="family deluxe"/>
                <Room name="double deluxe"/>
                <Room name="single deluxe"/>
            </div>
        </div>
    );
}

export default FeaturedRooms;