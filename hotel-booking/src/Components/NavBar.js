import React from "react";
import logo from "../images/logo.svg";
import {FaAlignRight} from "react-icons/fa";
import {Link} from "react-router-dom";

class NavBar extends React.Component {
    state = {
        isOpen: false
    }

    ButtonToggle = () => {
        if(!this.state.isOpen) {
            this.setState({isOpen: true});
        }
        else {
            this.setState({isOpen: false});
        }
    }

    render() {
        return(
            <nav className="navbar">
                <div className="nav-center">
                    <div className="nav-header">
                        <Link to="/">
                            <img src={logo} alt=""></img>
                        </Link>
                        <button className="nav-btn" onClick={this.ButtonToggle}>
                            <FaAlignRight className="nav-icon"/>
                        </button>
                    </div>
                    <ul className={this.state.isOpen ? "nav-links show-nav" : "nav-links"}>
                        <li>
                            <Link to="/">HOME</Link>
                        </li>
                        <li>
                            <Link to="/rooms">ROOMS</Link>
                        </li>
                    </ul>
                </div>
            </nav>
        );
    }
}

export default NavBar;