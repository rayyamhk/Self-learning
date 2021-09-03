import React from "react";

import Hero from "../Components/Hero";
import Banner from "../Components/Banner";
import Title from "../Components/Title";
import RoomsList from "../Components/RoomsList";
import data from "../data";
import {FetchRoomByType, FetchRoomByCapacity, FetchRoomByPrice, FetchRoomBySize, FetchRoomByBreakfast, FetchRoomByPets} from "../Components/FetchRoom";
import {Link} from "react-router-dom";

class Rooms extends React.Component {
    state = {
        type: "all",
        guests: 1,
        price: 600,
        sizeLower: 1,
        sizeUpper: 3000,
        breakfast: false,
        pets: false,
    }

    Filter(type, guests, price, sizeLower, sizeUpper, breakfast, pets) {
        let temp = data;
        temp = FetchRoomByType(temp, type);
        temp = FetchRoomByCapacity(temp, guests);
        temp = FetchRoomByPrice(temp, price);
        temp = FetchRoomBySize(temp, sizeLower, sizeUpper);
        temp = FetchRoomByBreakfast(temp, breakfast);
        temp = FetchRoomByPets(temp, pets);
        console.log(temp);
        return temp;
    }

    render() {
        return(
            <div>
                <Hero name="roomsHero">
                    <Banner title="Our rooms" description="">
                        <Link className="btn-primary" to="/">return home</Link>
                    </Banner>
                </Hero>
                <div className="filter-container">
                    <Title title="search rooms"></Title>
                    <form className="filter-form">
                        <div className="form-group">
                            <label>room type</label>
                            <select className="form-control" name="room-type" value={this.state.type} onChange={(e) => {
                                this.setState({type: e.target.value})
                            }}>
                                <option value="all">all</option>
                                <option value="family">family</option>
                                <option value="single">single</option>
                                <option value="double">double</option>
                                <option value="triple">triple</option>
                                <option value="presidential">presidential</option>
                            </select>
                        </div>
                        <div className="form-group">
                            <label>guests</label>
                            <select className="form-control" name="guests" value={this.state.guests} onChange={(e) => {
                                this.setState({guests: e.target.value})
                            }}>
                                <option value="1">1</option>
                                <option value="2">2</option>
                                <option value="3">3</option>
                                <option value="4">4</option>
                                <option value="5">5</option>
                                <option value="6">6</option>
                            </select>
                        </div>
                        <div className="form-group">
                            <label>room price ${this.state.price}</label>
                            <input type="range" name="price" min="0" max="600" value={this.state.price} onChange={(e) => {
                                this.setState({price: e.target.value})
                            }}></input>
                        </div>
                        <div className="form-group">
                            <label>room size</label>
                            <div className="size-inputs">
                                <input type="number" value={this.state.sizeLower} className="size-input" onChange={(e) => {
                                    this.setState({sizeLower: parseInt(e.target.value)})
                                }}/>
                                <input type="number" value={this.state.sizeUpper} className="size-input" onChange={(e) => {
                                    this.setState({sizeUpper: parseInt(e.target.value)})
                                }}/>
                            </div>
                        </div>
                        <div className="form-group">
                            <div className="single-extra">
                                <input type="checkbox" name="breakfast" onClick={(e) => {
                                    this.setState({breakfast: e.target.checked ? true : false});
                                }}/>
                                <label>breakfast</label>
                            </div>
                            <div className="single-extra">
                                <input type="checkbox" name="pets" onClick={(e) => {
                                    this.setState({pets: e.target.checked ? true : false});
                                }}/>
                                <label>pets</label>
                            </div>
                        </div>
                    </form>
                </div>
                <RoomsList roomsDisplayed={this.Filter(this.state.type, this.state.guests, this.state.price, this.state.sizeLower, this.state.sizeUpper, this.state.breakfast, this.state.pets)} />
            </div>
        )
    }
}

export default Rooms