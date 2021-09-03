import React from 'react';
import Button from './Buttons/Button';
import SocialButton from './Buttons/SocialButton';
import FontAwesomeButton from './Buttons/FontAwesomeButton';

interface Props {
  fontawesome: string;
  href?: string;
  color?: string;
  backgroundColor?: string;
  opacity?: string;
  size?: string;
  rippleColor?: string;
}

const App = () => {
  return(
    <>
    <Button text="Click Me" fontSize="12px" color="#ffd32a" backgroundColor="#000000" padding="12px 40px" border="40px" rippleColor="white"/>
    <Button text="TRANSPARENT" color="#ff3f34" padding="12px 18px" backgroundColor="#ffffff" opacity="0" rippleColor="#ff3f34" border="2px"/>
    <Button text="PRIMARY" rippleColor="#000"/>

    <Button text="1" margin="0px" border="999px 0 0 999px" rippleColor="#000"/>
    <Button text="2" margin="0px" border="0" rippleColor="#000"/>
    <Button text="3" margin="0px" border="0" rippleColor="#000"/>
    <Button text="4" margin="0px" border="0 999px 999px 0" rippleColor="#000"/>

    <SocialButton brand="youtube" size="10rem" border="30px" rippleColor="#000000"/>

    <FontAwesomeButton fontawesome="FaTwitter" color="#55ACEF" backgroundColor="#ffffff" opacity="0" size="10rem" rippleColor="#55ACEF" border="50% 0 0 50%"/>
    </>
  )
}

export default App;