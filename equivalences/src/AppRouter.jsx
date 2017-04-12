import React from 'react';
import {
    BrowserRouter as Router,
    Route
} from 'react-router-dom';

import SuggestedRulesList from 'containers/SuggestedRulesList';

function AppRouter() {
    return (
        <Router>
            <div className="ts text container" style={{ margin: '1rem' }}>
                <Route
                  exact
                  path="/"
                  component={SuggestedRulesList}
                />
            </div>
        </Router>
    );
}

export default AppRouter;
