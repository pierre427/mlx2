# Calculator licensing decision

On 17 November 2026, Pierre Lamy requested free noncommercial use, attribution,
no commercial use, and no distribution of derivatives. He confirmed Pierre
Lamy as the copyright holder. The existing project contact is
`pierre@userid.org`.

The [license](../../provenance/LICENSE-calculator.txt) implements that policy
as a custom source-available license. [Exact scope](../../provenance/LICENSE-calculator-scope.json)
identifies dedicated calculator files and the calculator portions of the
shared deterministic-tile controller. Generic infrastructure, fact tables,
dependencies and model weights keep their existing terms. The root Apache
license remains intact; README, NOTICE, file headers and package metadata
identify the scoped exception. The package uses an AND expression, not an OR
choice that would offer the calculator under Apache as an alternative.

Private noncommercial modifications are allowed. Sharing is limited to
unchanged source copies or unchanged binaries supplied by the copyright
holder, with notices. Commercial lab research, evaluation, model training,
fine-tuning, distillation and deployment require a separate written license,
even if no product has yet earned revenue. Attribution is required in public
reports of results, without adding a credit to every numerical answer.

The public-export tool refuses an export referencing these terms unless the
license, scope and NOTICE are explicitly selected together. No public export
or publication was performed. Read-only inspection of public `main` at
`71204c352a79bdcc6f7f84431b3c25e04a63b32a` found no calculator runtime files;
this is not an audit of every fork, earlier distribution or external copy.

## Limits to the protection

The prior private source at `fe509e6142154544e35f058f155ea09d49c1113d` carried
Apache-2.0 project terms. The new license expressly preserves rights already
validly granted under earlier copies. Apache's copyright grant is irrevocable;
changing a notice cannot withdraw it. [Apache License 2.0, sections 2 and 4](https://www.apache.org/licenses/LICENSE-2.0).

Copyright can protect the code's original expression, but does not establish
ownership of an algorithm or prohibit an independently written implementation
of an idea. This work has not received a patentability or ownership opinion.
[Canadian Intellectual Property Office: copyright guide](https://ised-isde.canada.ca/site/canadian-intellectual-property-office/en/guide-copyright).

The standard PolyForm Noncommercial license permits distribution of modified
works, while PolyForm Strict removes permission to modify and distribute and
has broad institutional permissions. Neither directly expresses this requested
policy. The custom license does not claim to be either PolyForm license.
[PolyForm Noncommercial](https://polyformproject.org/licenses/noncommercial/1.0.0),
[PolyForm Strict](https://polyformproject.org/licenses/strict/1.0.0).

These are applied project terms, not a lawyer's opinion on enforceability or
exclusive ownership. Have an IP lawyer review the custom terms and prior
distribution history before relying on them in a commercial dispute or
negotiating a frontier-lab license. No registration or patent filing was made.
