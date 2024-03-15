import { Component, TemplateRef, ViewChild, ElementRef, AfterViewChecked } from '@angular/core';
import { NbDialogService } from '@nebular/theme';
import { TermDialogComponent } from './components/term-dialog.component';
import { HttpClient } from '@angular/common/http';
import { environment } from '../environments/environment';


@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  title = "IA Legal Argentina";

  userInput: string = "";
  inputCheckbox: boolean = false;
  response: any = [];
  references: any = [];
  displayReferences: any = [];
  message: string = "";
  question: boolean = true;
  loading: boolean = true;
  refer: boolean = false;

  constructor(private dialogService: NbDialogService, private httpClient: HttpClient) {  }
  @ViewChild('resp') typewriterElement!: ElementRef;

  onEnter() {
    console.log("Pregunta:", this.userInput);
    if (this.inputCheckbox == false) {
      this.message = "Deben aceptarse los términos y condiciones";
      this.openError();
    } else {
      if (this.userInput == "") {
        this.message = "Debe ingresar alguna pregunta";
        this.openError();
      } else {
        this.question = false;

        const url = environment.url;
        const url_question = url + "/question";
        console.log("URL", url_question);

        this.httpClient
          .post(url_question, {'question': this.userInput})
          .subscribe({
            next: this.responseBot,
            error: err => console.error('Ops: ', err.message),
            complete: () => console.log('Completed Contract Info')
          });
      }
    }
  }

  private responseBot = (data: any): any => {
    console.log('Response', data);
    this.loading = false;
    this.response = [data.answer];
    this.references = data.sources;
  }

  handleComplete() {
    this.refer = true;
    this.references.forEach((item: any, index: any) => {
      setTimeout(() => {
        this.displayReferences.push(item);
      }, 1000 * index);
    });
  }

  newQuestion() {
    this.loading = true;
    this.question = true;
    this.refer = false;
    this.userInput = "";
    this.response = [];
    this.references = [];
    this.displayReferences = [];
  }

  openDialog(dialog: TemplateRef<any>) {
      this.dialogService.open(dialog);
  }

  openError() {
    this.dialogService.open(TermDialogComponent, {
      context: {
        message: this.message,
      },
    });
  }

}
