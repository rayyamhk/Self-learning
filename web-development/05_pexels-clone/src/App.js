import React from 'react';
import Navbar from "./Components/Navbar";
import SubNavbar from "./Components/SubNavbar";
import ImageShowcase from "./Components/ImageShowcase";
import Overlay from "./Components/Overlay";

class App extends React.Component {
    state = {
        path: "photos",
        image: {
            id: "",
            url: "",
            author: ""
        },
        isOpen: false,
        showLike: false,
        windowWidth: window.innerWidth
    }

    componentDidMount() {
        window.addEventListener('resize', () => {
            this.setState({windowWidth:window.innerWidth});
        });
    }
      
    componentWillUnmount() {
        window.removeEventListener('resize', () => {
            this.setState({windowWidth:window.innerWidth});
        });
    }

    render() { 
        return ( 
            <div className="page-wrappper">
                <Navbar/>
                <SubNavbar path={this.state.path} subnavToggle={this.subnavToggle}/>
                <Overlay image={this.state.image} isOpen={this.state.isOpen} toggle={this.setIsOpen}/>
                <ImageShowcase path={this.state.path} width={this.state.windowWidth} showLike={this.state.showLike} imgToggle={this.imgToggle} likeShow={this.likeShow} likeHide={this.likeHide}/>
            </div>
        );
    }

    subnavToggle = (path) => {
        if(this.state.path !== path) {
            this.setState({path:path});
        }
    }
    imgToggle = (img) => {
        this.setState({ image : img});
        this.setState({isOpen:!this.state.isOpen});
    }
    setIsOpen = () => {
        this.setState({isOpen:!this.state.isOpen});
    }
    likeShow = () => {
        this.setState({showLike: true});
    }
    likeHide = () => {
        this.setState({showLike: false});
    }
}
 
export default App;