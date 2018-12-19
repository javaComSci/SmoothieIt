import Fruit from './Fruit'
'use strict';
const e = React.createElement;

class App extends React.Component {
  constructor(props) {
    super(props);
  }

  render() {
    return e(
      'div', {}, 'REACT CODE'
    );
  }
}

const domContainer = document.querySelector('#app_container');
ReactDOM.render(e(App), domContainer);


 // class Greet extends React.Component {
 //          render() {
 //              return (<div><App/><div>);
 //          }
 //      }
 //      ReactDOM.render(
 //          <Greet />,
 //          document.getElementById('root')
 //      );