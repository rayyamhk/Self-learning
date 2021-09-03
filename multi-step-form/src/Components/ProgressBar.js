import React from "react";
import ProgressHead from "./ProgressHead";
import ProgressBody from "./ProgressBody";
import ProgressTail from "./ProgressTail";

class ProgressBar extends React.Component{

    render() {
        let progressBars = [];
        let descriptions = [
            "please input your name and email",
            "basic info about you",
            "almost finish",
            "review"
        ]
        
        for(let i = 0; i < this.props.total; i++) {
            if(i === 0) {
                if(i < this.props.step-1) {
                    progressBars.push(<ProgressHead key={i+1} step={i+1} description={descriptions[i]} status={"finished"}/>);
                }
                else if(i === this.props.step-1) {
                    progressBars.push(<ProgressHead key={i+1} step={i+1} description={descriptions[i]} status={"processing"}/>);
                }
                else {
                    progressBars.push(<ProgressHead key={i+1} step={i+1} description={descriptions[i]} status={"not-reached"}/>);
                }
            }
            else if(i === this.props.total - 1){
                if(i < this.props.step-1) {
                    progressBars.push(<ProgressTail key={i+1} step={i+1} description={descriptions[i]} status={"finished"}/>);
                }
                else if(i === this.props.step-1) {
                    progressBars.push(<ProgressTail key={i+1} step={i+1} description={descriptions[i]} status={"processing"}/>);
                }
                else {
                    progressBars.push(<ProgressTail key={i+1} step={i+1} description={descriptions[i]} status={"not-reached"}/>);
                }
            }
            else {
                if(i < this.props.step-1) {
                    progressBars.push(<ProgressBody key={i+1} step={i+1} description={descriptions[i]} status={"finished"}/>);
                }
                else if(i === this.props.step-1) {
                    progressBars.push(<ProgressBody key={i+1} step={i+1} description={descriptions[i]} status={"processing"}/>);
                }
                else {
                    progressBars.push(<ProgressBody key={i+1} step={i+1} description={descriptions[i]} status={"not-reached"}/>);
                }
            }
        }

        return(
            <div className="progressbar-container">
                {progressBars}
            </div>
        );
    }
}

export default ProgressBar;