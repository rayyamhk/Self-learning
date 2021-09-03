import React from "react";
import Room from "./Room";

function RoomList({roomsDisplayed}) {
    if(roomsDisplayed.length !== 0) {
        return(
            <div className="roomslist">
                <div className="roomslist-center">
                    {
                        roomsDisplayed.map((room, index) => 
                            <Room key={index} name={room.fields.name}/>
                        )
                    }
                </div>
            </div>
        )
    }
    else {
        return(
            <div className="empty-search">
                <h3>Sorry, no rooms matched</h3>
            </div>
        )
    }
}

export default RoomList