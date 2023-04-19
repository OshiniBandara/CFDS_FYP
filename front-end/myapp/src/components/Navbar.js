import React, {useState} from "react";
import { Link } from "react-router-dom";
import { NavLink } from "react-router-dom";
import "./Navbar.css";
import {BsSlack} from "react-icons/bs";
import {FaBars, FaTimes } from "react-icons/fa";
import {IconContext} from "react-icons/lib";

function Navbar() {

    const [click, setClick] = useState(false)

    const handleClick = () => setClick(!click)
    const closeMobileMenu = () => setClick(false)
     
    return (
        <>
        <IconContext.Provider value={{color: "#09bef0"}}>
            <div className="navbar">
                <div className="navbar-container container">
                    <Link to="/" className="navbar-logo" onClick={closeMobileMenu}>
                        <BsSlack className="navbar-icon" />
                        CFDS
                    </Link>
                    <div className="menu-icon" onClick={handleClick}>
                        {click ? <FaTimes /> : <FaBars/> }
                    </div>
                    <ul className={click ? "nav-menu active" : "nav-menu"}>
                        <li className="nav-item">
                            <NavLink to="/" className={({isActive}) => "nav-links" + (isActive ? " activated" : "")} onClick={closeMobileMenu}>
                                Home
                            </NavLink>
                        </li>
                        <li className="nav-item">
                            <NavLink to="/detector" className={({isActive}) => "nav-links" + (isActive ? " activated" : "")} onClick={closeMobileMenu}>
                                Detector
                            </NavLink>
                        </li>
                        <li className="nav-item">
                            <NavLink to="/experiments" className={({isActive}) => "nav-links" + (isActive ? " activated" : "")} onClick={closeMobileMenu}>
                                Experiments
                            </NavLink>
                        </li>
                        <li className="nav-item">
                            <NavLink to="/help" className={({isActive}) => "nav-links" + (isActive ? " activated" : "")} onClick={closeMobileMenu}>
                                Help
                            </NavLink>
                        </li>
                    </ul>
                </div>
            </div>
            </IconContext.Provider>
        </>
    );
}

export default Navbar;
