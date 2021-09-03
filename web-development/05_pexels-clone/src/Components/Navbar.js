import React, {useState} from 'react';
import logo from "../img/logo.jpg";

function Navbar() {
    const [isOpen, setIsOpen] = useState(false);
    return(
        <nav>
            <div className="logo-container">
                <span className="logo">
                    <a href="/"><img src={logo} alt="logo"></img></a>
                </span>
                <span className="logo-name">Pexels</span>
            </div>
            <div className="search-bar">
                <input type="text"></input>
                <i className="fas fa-search"></i>
            </div>
            <div className="nav-btn" onClick={() => setIsOpen(!isOpen)}>
                <i className="fas fa-bars"></i>
            </div>
            <ul className={isOpen?"nav-open nav-close":"nav-close"}>
                <li><a href="/">login</a></li>
                <li><a href="/">join</a></li>
                <li><a href="/">explore</a></li>
            </ul>
            <div className="ellipsis-container">
                <i className="fas fa-ellipsis-h"></i>
            </div>
            <div className="upload-btn">
                <span>Upload</span>
            </div>
        </nav>
    )
}

export default Navbar;