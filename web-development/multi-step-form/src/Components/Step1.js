import React from "react";

function Step1({info, setInfo, errorMsg}) {
    return(
        <form>
            {
                errorMsg !== "" && <div className="warning">{errorMsg}</div>
            }
            <div className="input-container">
                <label>first name</label>
                <input type="text" name="firstname" value={info.firstname} required onChange={(e) => {
                    setInfo(e);
                }}></input>
            </div>
            <div className="input-container">
                <label>last name</label>
                <input type="text" name="lastname" value={info.lastname} required onChange={(e) => {
                    setInfo(e);
                }}></input>
            </div>
            <div className="input-container">
                <label>email</label>
                <input type="email" name="email" value={info.email} required onChange={(e) => {
                    setInfo(e);
                }}></input>
            </div>
        </form>
    );
}

export default Step1;