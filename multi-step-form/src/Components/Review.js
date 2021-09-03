import React from "react";

class Review extends React.Component {
    render() {
        const {info} = this.props;
        let temp = [];
        Object.keys(info).forEach((key,index) => {
            if(key !== "accept") {
                console.log(key);
                let answer = info[key] || "None";
                temp.push(
                    <>
                        <div className="review-row" key={index}>
                            <p className="review-question">{key}</p>
                            <p className="review-answer">{answer}</p>
                        </div>
                    </>
                )
            }
        })
        return(
            <>
                <div className="review-container">
                    {temp}
                </div>
            </>
        );
    }
}

export default Review;