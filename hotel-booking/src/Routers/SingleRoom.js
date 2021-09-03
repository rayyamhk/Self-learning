import React from "react";
import Hero from "../Components/Hero";
import Banner from "../Components/Banner";
import data from "../data";
import {FetchRoomByPath} from "../Components/FetchRoom";
import {Link} from "react-router-dom";

function SingleRoom({match}) {
    let path = match.params.path;
    let roomDetails = FetchRoomByPath(data, path);
    let roomImages = roomDetails.fields.images;
    if(roomDetails) {
        return(
            <div>
                <Hero name="roomsHero" backgroundImg={roomImages[0].fields.file.url}>
                    <Banner title={roomDetails.fields.name} description="">
                        <Link className="btn-primary" to="/rooms">back to rooms</Link>
                    </Banner>
                </Hero>
                <div className="single-room">
                    <div className="single-room-images">
                        <img src={roomImages[1].fields.file.url} alt=""></img>
                        <img src={roomImages[2].fields.file.url} alt=""></img>
                        <img src={roomImages[3].fields.file.url} alt=""></img>
                    </div>
                    <div className="single-room-info">
                        <div className="desc">
                            <h3>details</h3>
                            <p>{roomDetails.fields.description}</p>
                        </div>
                        <div className="info">
                            <h3>info</h3>
                            <h6>Price : ${roomDetails.fields.price}</h6>
                            <h6>Size : {roomDetails.fields.size} sqft</h6>
                            <h6>Max Capacity : {roomDetails.fields.capacity} People</h6>
                            <h6>Pets {roomDetails.fields.pets ? "Allowed" : "Not Allowed"}</h6>
                            <h6>Free Breakfast {roomDetails.fields.breakfast ? "Included" : "Not Included"}</h6>
                        </div>
                    </div>
                    <div className="room-extras">
                        <h6>Extras</h6>
                        <ul className="extras">
                            {roomDetails.fields.extras.map((extra, index) => 
                                <li key={index}>{extra}</li>
                            )}
                        </ul>
                    </div>
                </div>
            </div>
        );
    }
    return(
        <h1>NULL</h1>
    );
}

export default SingleRoom;