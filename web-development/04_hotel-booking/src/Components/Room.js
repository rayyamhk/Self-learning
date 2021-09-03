import React from "react";
import data from "../data";
import {FetchRoomByName} from "./FetchRoom";
import {Link} from "react-router-dom";


function Room({name}) {
    let roomDetails = FetchRoomByName(data, name);
    if(roomDetails) {
        let path = "/rooms/" + roomDetails.fields.slug;
        return(
            <div className="room">
                <div className="img-container">
                    <img src={roomDetails.fields.images[0].fields.file.url} alt=""></img>
                    <div className="price-top">
                        <h6>${roomDetails.fields.price}</h6>
                        <p>per night</p>
                    </div>
                    <Link to={path} className="room-link btn-primary">features</Link>
                </div>
                
                <div className="room-info">{name}</div>
            </div>
        );
    }
    return(
        <h1>NULL</h1>
    );
}

export default Room;