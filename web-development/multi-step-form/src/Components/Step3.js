import React from "react";

function Step3({info, setInfo, errorMsg}) {
    return(
        <form>
            {
                errorMsg !== "" && <div className="warning">{errorMsg}</div>
            }
            <div className="input-container">
                <label>about you (Optional) </label>
                <textarea rows="4" cols="50" name="message" value={info.message} onChange={(e) => {
                    setInfo(e);
                }}></textarea>
            </div>      
            <div className="input-container">
                <label>where did you hear about us?</label>
                <select name="source" value={info.source} onChange={(e) => {
                    setInfo(e);
                }}>
                    <option value=""></option>
                    <option value="search engine">search engine</option>
                    <option value="social media">social media</option>
                    <option value="linkedin">LinkedIn</option>
                    <option value="recommendation">recommendation</option>
                    <option value="other">other</option>
                </select>
            </div>

            <div className="input-container">
                <div className="checkbox-container">
                    <input type="checkbox" name="accept" value={info.accept === "" ? "checked" : ""} onChange={(e) => {
                        setInfo(e);
                    }}/>
                    <label>I accept the term of Service</label>
                </div>
            </div>      
        </form>
    );
}

export default Step3;